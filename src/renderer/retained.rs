//! A viewport's copy of its world's retained instances (render-engine#95; Bevy's retained
//! render world, where a mesh entity's render data persists and only changed entities are
//! extracted again).
//!
//! The instance buffer's first rows mirror the world's instance slots, one row per slot. Each
//! `prepare` derives again only the slots edited since the last one (and those whose mesh or
//! material changed), writes those rows alone, and re-sorts the retained instances only when
//! one of them changed its draw (mesh, material, phase or face). Blended instances are ordered
//! by depth and skinned ones posed every frame, as on the listed path.

use super::{
    gpu::InstanceData,
    prepare::derive_instance,
    viewport::{DrawKey, Growable, Phase, PrepareWork},
};
use crate::{cull::Obb, world::World};
use bytemuck::Zeroable;

/// Rows closer than this are written together with the clean rows between them, in one write.
const MERGE_GAP: usize = 16;

#[derive(Default)]
pub(super) struct RetainedInstances {
    /// The world (its id) and its tick this copy was derived at; `None` before the first sync.
    synced: Option<(u64, u64)>,
    /// Per slot: the instance's key (`view_z` unset), `None` for an empty slot.
    keys: Vec<Option<DrawKey>>,
    /// Per slot: the instance's row as written to the GPU (zeroed for an empty slot, so the
    /// GPU culling pass skips it).
    pub(super) data: Vec<InstanceData>,
    /// Per slot: the instance's box (skinned instances: boxed each frame).
    pub(super) obbs: Vec<Obb>,
    /// The opaque and alpha-masked instances' keys in draw order.
    pub(super) sorted: Vec<DrawKey>,
    /// Blended instances' slots, ordered by depth each frame.
    pub(super) blended: Vec<u32>,
    /// Skinned instances' slots, posed and boxed each frame.
    pub(super) skinned: Vec<u32>,
    /// Whether a key changed since `sorted`, `blended` and `skinned` were built.
    resort: bool,
    /// Slots whose GPU row is stale.
    pub(super) dirty: Vec<u32>,
    /// Every row is stale (another world, or a new instance buffer).
    all_dirty: bool,
}

impl RetainedInstances {
    /// Rows of the instance buffer the retained instances take: one per world slot.
    pub(super) fn rows(&self) -> u32 {
        self.keys.len() as u32
    }

    /// A live slot's key.
    pub(super) fn key(&self, slot: u32) -> DrawKey {
        self.keys[slot as usize].expect("a live slot")
    }

    /// Retained instances drawn.
    pub(super) fn count(&self) -> u32 {
        (self.sorted.len() + self.blended.len()) as u32
    }

    /// Derives what the world changed since the last sync.
    pub(super) fn sync(&mut self, world: &World) {
        let since = match self.synced {
            Some((id, tick)) if id == world.id() => {
                if tick == world.tick() {
                    return;
                }
                tick
            }
            _ => {
                *self = Self {
                    resort: true,
                    all_dirty: true,
                    ..Self::default()
                };
                0
            }
        };
        let slots = world.instance_slot_count();
        self.keys.resize(slots, None);
        self.data.resize(slots, InstanceData::zeroed());
        self.obbs.resize(slots, Obb::default());
        let content = world.content_tick() > since;
        for (slot, (instance, tick)) in world.instance_slots().enumerate() {
            let stale = tick > since
                || instance.is_some_and(|i| {
                    content
                        && (world.mesh(i.mesh).changed > since
                            || world.material(i.material).changed > since)
                });
            if !stale {
                continue;
            }
            let old = self.keys[slot];
            match instance {
                None => {
                    self.keys[slot] = None;
                    self.data[slot] = InstanceData::zeroed();
                }
                Some(i) => {
                    // Skinned instances get their joints and box each frame.
                    let (key, obb, data) = derive_instance(world, i, slot as u32, None);
                    self.keys[slot] = Some(key);
                    self.obbs[slot] = obb;
                    self.data[slot] = data;
                }
            }
            self.resort |= match (old, self.keys[slot]) {
                (Some(a), Some(b)) => !a.same_draw(&b),
                (None, None) => false,
                _ => true,
            };
            self.dirty.push(slot as u32);
        }
        self.synced = Some((world.id(), world.tick()));
        if std::mem::take(&mut self.resort) {
            self.sorted.clear();
            self.blended.clear();
            self.skinned.clear();
            for (slot, key) in self.keys.iter().enumerate() {
                let Some(key) = key else { continue };
                if key.skinned {
                    self.skinned.push(slot as u32);
                }
                if key.phase == Phase::Blend {
                    self.blended.push(slot as u32);
                } else {
                    self.sorted.push(*key);
                }
            }
            self.sorted.sort_unstable_by(DrawKey::draw_order);
        }
    }

    /// Sets a row's GPU culling row (`flags.w`), marking it stale when it changes.
    pub(super) fn set_gpu_row(&mut self, slot: u32, row_plus_one: u32) {
        let flags = &mut self.data[slot as usize].flags;
        if flags.w != row_plus_one {
            flags.w = row_plus_one;
            self.dirty.push(slot);
        }
    }

    /// Writes the stale rows into the start of `buffer`; `replaced` says the buffer is new.
    /// Returns the rows written.
    pub(super) fn upload(
        &mut self,
        buffer: &Growable,
        replaced: bool,
        queue: &wgpu::Queue,
        work: &mut PrepareWork,
    ) -> u32 {
        let stride = size_of::<InstanceData>() as u64;
        let mut write = |rows: std::ops::Range<usize>| {
            let bytes: &[u8] = bytemuck::cast_slice(&self.data[rows.clone()]);
            queue.write_buffer(&buffer.buffer, rows.start as u64 * stride, bytes);
            work.upload_bytes += bytes.len() as u64;
        };
        let written = if std::mem::take(&mut self.all_dirty) || replaced {
            if !self.data.is_empty() {
                write(0..self.data.len());
            }
            self.data.len() as u32
        } else {
            self.dirty.sort_unstable();
            self.dirty.dedup();
            let mut run: Option<std::ops::Range<usize>> = None;
            for &slot in &self.dirty {
                let slot = slot as usize;
                run = match run {
                    Some(r) if slot <= r.end + MERGE_GAP => Some(r.start..slot + 1),
                    Some(r) => {
                        write(r);
                        Some(slot..slot + 1)
                    }
                    None => Some(slot..slot + 1),
                };
            }
            if let Some(r) = run {
                write(r);
            }
            self.dirty.len() as u32
        };
        self.dirty.clear();
        written
    }

    /// CPU memory this copy holds.
    pub(super) fn memory(&self) -> u64 {
        (self.keys.capacity() * size_of::<Option<DrawKey>>()
            + self.data.capacity() * size_of::<InstanceData>()
            + self.obbs.capacity() * size_of::<Obb>()
            + self.sorted.capacity() * size_of::<DrawKey>()
            + self.blended.capacity() * size_of::<u32>()
            + self.skinned.capacity() * size_of::<u32>()
            + self.dirty.capacity() * size_of::<u32>()) as u64
    }
}

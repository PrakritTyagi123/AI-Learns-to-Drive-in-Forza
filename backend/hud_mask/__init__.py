"""ForzaTek AI v2 — Module 3: HUD Mask.

The user paints over the static in-game HUD (minimap, speedometer, gear,
lap counter, etc.) on ONE sample frame per game version. We store that mask
once, keyed by game_version, and apply it at read-time wherever frames are
consumed downstream — labeling, training dataset, prediction overlays.

Frames on disk are NEVER modified. The mask is a multiplicative alpha
applied lazily by `auto_propagate.apply_mask()`. This means:

    * Re-painting the mask updates every existing frame instantly.
    * Switching from FH4 to FH5 uses a different mask automatically.
    * Resolution changes don't break anything — rectangles are stored
      in normalized 0..1 coordinates.

Public surface:
    service.save_mask(game_version, rects, sample_frame_id)
    service.get_mask(game_version)
    service.list_masks()
    service.delete_mask(game_version)
    auto_propagate.apply_mask(image_bgr, game_version)
    auto_propagate.get_mask_array(game_version, h, w)

This module reads from Module 2's `frames` table to fetch sample frames
but never writes to it. It only writes to the `hud_masks` table.
"""
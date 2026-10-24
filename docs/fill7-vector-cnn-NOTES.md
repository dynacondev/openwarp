# Fill #7 — Tiny infilling-vector CNN (future work, NOT scheduled)

Copy-donor `(a,b)` predictor (Kanchana et al. WACV22 IVP distilled), NOT color regression,
NOT LaMa / M2SVid / ExtraSSNet / GFFE-SCN at runtime.

## Why deferred
- #1–#4 (spatial heuristic) + #5/#6 (temporal cache) hit ~1ms at 1024² with no training.
- #7 needs dataset + training + Vulkan inference plumbing. Weeks, plus eval harness.

## What to build when revisited
1. **Dataset from our own warp:** inputs per hole pixel = warped color + warped depth
   (`dn+1^w`) + hole mask + warped temporal prior `(α^w,β^w)` (warp prior vectors by
   camera delta, box-blur denoise — IVP Sec 3.2). Targets = exhaustive BG search
   `(a,b)` (nearest valid with `depth>=bgDepth-TOL`, min 3x3 SAD). Mine from
   `openwarp-vulkan-test` synthetic + demo dumps (`OPENWARP_VK_DUMP=1` PPM+mask).
2. **Model:** 2–3 layer `3x3 d=2,4` dilated CNN predicting 2ch `(a,b)` on holes only.
   No FFC/GAN/diffusion. Quantized int8. Loss `L=MSE_masked + SSIM + 0.001*smoothness`
   (IVP), backprop through sampler `fi(x,y)=fw(x+a,y+b)`.
3. **Runtime:** new `warp_infer.comp` (or ONNX-RT/NCNN) + weight UBOs + history buffers.
   Vulkan 1.0 compute has no ML ops — hand-port or external infer lib required.
   Fallback to #2 directional march when confidence low / holes wide.
4. **Test-time iterative correction (IVP P=3):** if vector lands on FG (depth test),
   zero it; infill BG-only holes, shrink, repeat; last iter extend until known.
5. **Acceptance:** same as heuristic engines (`blackFrac<0.15`, valid bit-exact,
   whiteness < mip baseline) + temporal-consistency metric (variance along
   foreground-offset track, IVP TC) + wide-mask eval (LaMa protocol: wide + box).

## References
- IVP: https://arxiv.org/abs/2110.08805
- LaMa (why not): https://github.com/advimman/lama
- M2SVid (sparse hole-only idea): https://arxiv.org/abs/2505.16565
- GFFE (focus mask, BG buffer): https://arxiv.org/abs/2406.18551
- ExtraSS (bilateral gather): SIGGRAPH Asia 2023 sess155 / papers_744
- Intel 12530742: reproject+weight+GAN claims — route learned blender/inpainter via counsel.

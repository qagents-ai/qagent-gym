# Quentin team artwork

## Animated Q gym

`quentin-gym-future-animated.webp` is a looping, locally rendered 3D illustration created
on 2026-11-19. The Q screen has orbiting lights and moving telemetry bars. The
robots run on a moving treadmill, lift a memory cube and wave at the evaluation
station. The scene uses ceramic shells, metallic surfaces and teal studio lighting.
The screen bars are decorative; they contain no measured results.

The display, robot badges and memory cube use the existing QAgents journal-Q
mark, with its periwinkle and mint colours for dark backgrounds. The large mark
gently pulses as the orbit lights move. `brand/qagents-icon-dark.svg` preserves
the original three-path geometry from QAgents' `assets/logo/qagents-icon.svg`
(commit `1e7bc6fd9fc78ef562b1ab1f49d86f961c39d6f7`); its PNG texture is packed into the generated Blender file.
The same SVG and a 32 px PNG appear in the private guide's header and favicon.

The README and RL guide select `quentin-gym-future.webp` when the browser requests
reduced motion. Otherwise, browsers load the animated WebP, with
`quentin-gym-future.gif` as a compatibility fallback. The three files share the
same composition and aspect ratio. Each animation has 72 frames with a 60 ms
delay, repeating every 4.32 seconds.

Rebuild using Blender 4.4+ and ImageMagick 7 from the repository root:

```powershell
magick -background none assets/brand/qagents-icon-dark.svg -resize 1024x1024 assets/brand/qagents-icon-dark.png
magick -background none assets/brand/qagents-icon-dark.svg -resize 32x32 assets/brand/qagents-icon-32.png
blender --background --threads 8 --python scripts/render_gym_animation.py -- --output .work/gym-animation
magick -delay 6 -loop 0 '.work/gym-animation/frame-*.png' -resize 800x500 -quality 85 -define webp:method=5 assets/quentin-gym-future-animated.webp
magick .work/gym-animation/frame-0001.png -resize 800x500 -colors 256 -unique-colors .work/gym-animation/palette.png
magick -delay 6 -loop 0 '.work/gym-animation/frame-*.png' -resize 800x500 +dither -remap .work/gym-animation/palette.png -layers Optimize assets/quentin-gym-future.gif
magick .work/gym-animation/frame-0001.png -quality 88 assets/quentin-gym-future.webp
```

The script saves an editable `.work/gym-animation/qagent-gym.blend` and all PNG
frames. Add `--preview` to render a single frame. Blender and ImageMagick are
artwork tools; neither is a Gym runtime dependency. Rendering uses Cycles with
12 samples and denoising. Hardware and Blender versions can affect pixel output.

## Earlier illustrations

Generated on 2026-11-13 using the built-in image_gen tool. Both images were inspected for readable labels, cheerful team imagery and faithful depiction of context learning. These are illustrations, not benchmark results or a product screenshot.

- `quentin-gym-hero.png`: friendly robot teammates practicing at memory, playbook, feedback and evaluation stations.
- `quentin-gym-learning-loop.png`: Try → Reflect → Remember → Evaluate, with evaluation keeping memory fixed.

- `quentin-rl-lab.webp` was the original RL guide illustration. Generated with the built-in
  Codex ImageGen tool on 2026-11-18, using the Gym hero as a character and style
  reference.

## Prompt 1 — hero

Use case: stylized-concept. Asset: landscape hero illustration for the Quentin team's open-source-style QAgent Gym project. Create a beautiful, happy, playful isometric miniature robot learning gym on a warm ivory background. Friendly rounded robot teammates with expressive simple faces and small Q emblems are practicing together: one robot carries colorful memory cards to a little library, two collaborate over an open playbook, one learns from a coach's feedback card, and a robot crosses a tidy evaluation obstacle course. Charming tactile 3D illustration with soft clay-like surfaces, crisp intentional details, warm daylight, cheerful but sophisticated design, generous breathing room around the scene. A coherent palette of teal, coral, warm yellow and soft lavender, with dark charcoal lettering. Prominent exact readable title: 'QAgent Gym'. Smaller exact subtitle: 'QUENTIN TEAM'. Four little station signs only: 'MEMORY', 'PLAYBOOK', 'FEEDBACK', 'EVALUATE'. Make this feel like a welcoming creative coding team celebrating practice and learning. Wide 16:9 composition, polished high resolution, no official OpenAI/Codex logo, no photorealistic people, no numerical benchmark claims, no random gibberish, no watermarks.

## Prompt 2 — learning loop

Use case: infographic-diagram. Asset: landscape illustrated explainer for the Quentin team's QAgent Gym. Create a joyful polished four-panel learning adventure with friendly small rounded robot teammates, in clean tactile 3D illustration with soft clay-like materials on warm ivory. Use teal, coral, warm yellow and soft lavender accents with charcoal lettering. Exact top title 'A day at QAgent Gym'. Four clearly numbered panels from left to right, connected by a friendly curved dotted route: '1 TRY' robot attempting a small task and making an innocent mistake; '2 REFLECT' two robot teammates thoughtfully discussing a feedback card; '3 REMEMBER' robot placing a useful lesson card in an open playbook beside a memory shelf; '4 EVALUATE' robot trying a DIFFERENT puzzle while the playbook sits closed and a friendly judge checks the result. Clear short captions, verbatim: 'Attempt a task', 'Learn from feedback', 'Keep useful lessons', 'Test without changing memory'. Footer text only: 'QUENTIN TEAM'. Express happiness, curiosity, teamwork and gentle humor, strong readability and generous whitespace, stylish editorial composition, wide 16:9 high-resolution output. No benchmark numbers, no claims of real model training or updating model weights, no official corporate logos, no random extra labels or watermarks.

## Prompt 3 — reinforcement-learning lab

Create a wide editorial banner illustration for the QAgent Gym guide on reinforcement learning. Match the charming tactile miniature clay style and the rounded white Quentin robots with black faceplates, teal ear caps and coral accents in the attached reference. Inside the same sunny training workshop, two Quentin robots test a small robot on a tabletop obstacle course. One robot reviews a compact scorecard with three colored marks and the other adjusts a set of brass calibration dials. Show the course and the test coming together as one clear scene, with a graceful loop in the track, a mint-teal workbench, and soft warm sunlight. Use a calm aqua and warm cream palette with coral details. Broad landscape composition, all important subjects in the central area, suitable for cropping into a short web banner. Keep the outer edges simple, no signage, no text, no labels, no new logos, no watermarks. This is a visual story for an engineering guide; technical diagrams are shown separately.

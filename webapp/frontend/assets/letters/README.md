Reference handshape diagrams for the reference panel (shown upfront in Learn
Mode, and after a miss in Letter/Word Mode).

## Source and license

These SVGs (`A.svg`–`Z.svg`) are from Wikimedia Commons,
[Category:ASL letters](https://commons.wikimedia.org/wiki/Category:ASL_letters),
originally created by wpclipart.com and explicitly released into the public
domain (CC0 — verified both on each file's Commons description page and in
the SVG's own embedded metadata). No attribution is legally required, though
crediting the source is good practice, which is why it's noted here.

J and Z are included even though the model doesn't classify them from a
static pose (they're recognized by motion — see `js/motionDetector.js`) —
the diagram still shows the resting handshape the motion starts from, and
`js/ghostAnimation.js`'s animated diagram demonstrates the actual movement.

## How this fits together

`app.js`'s `showReferenceFor()` tries `assets/letters/<LETTER>.svg` first; if
a letter's file is missing (or fails to load), the `<img>` just hides itself
and the ghost-hand skeleton diagram (drawn from the training data's mean
landmark positions, see `js/handTracking.js`'s `drawGhostHand`) carries the
teaching visual on its own. Either one alone is enough for the game to be
fully playable — this is a visual upgrade layered on top of a diagram that
was already real and working, not a required dependency.

## Replacing these

If you'd rather use different images (real photos, a different chart,
etc.), just overwrite the files here with the same names
(`A.svg`/`A.png`/etc. — any format an `<img>` tag can render). Double-check
the license of whatever you use before committing it, especially for a
public repo — see the note in the project's session history about why these
specific ones were chosen (verified public domain, not just "free to view").

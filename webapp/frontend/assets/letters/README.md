Optional reference images for the "miss" feedback panel.

Drop an image per letter here, named to match the letter exactly:

```
A.png  B.png  C.png  D.png  E.png  F.png  G.png  H.png  I.png
K.png  L.png  M.png  N.png  O.png  P.png  Q.png  R.png  S.png
T.png  U.png  V.png  W.png  X.png  Y.png
```

(No J or Z — the model only recognizes static letters, and J/Z require motion.)

If a letter's image is missing, the game automatically falls back to a short
text description of the handshape (see `LETTER_CUES` in `app.js`), so the game
is fully playable without adding any images at all.

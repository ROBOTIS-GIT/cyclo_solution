# FFW cuMotion model

These XRDF files describe the common FFW lift, arms, grippers, and `base_link`
collision model. Mobile-base wheel joints and wheel links are not part of the
planning robot model.

The group router selects an XRDF whose c-space matches the active MoveIt group:
`arm_l`, `arm_r`, `both_arms`, or `wholebody`. Before each request it copies the
latest available positions of non-c-space joints into the XRDF and reloads the
cuMotion robot model when required. Stored default positions are used only when
a live joint value is unavailable.

The collision sphere centers and radii originate from the supplied
`ai_worker.xrdf` (SHA-256
`c46a497a24b6874b3ddbd69ec0e00551729868fa80627392b66b2156cc03d300`).

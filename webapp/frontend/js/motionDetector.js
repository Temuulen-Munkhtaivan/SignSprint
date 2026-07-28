    const INDEX_TIP = 8;
    const PINKY_TIP = 20;

    const MAX_MOTION_TIME_MS = 3000;
    const MIN_FRAME_MOVEMENT = 0.004;

    let activeLetter = null;
    let startTime = 0;

    let phase = 0;
    let phaseStart = null;
    let previousPoint = null;
    let firstDirection = 0;

    export function resetMotionDetector(letter = null) {
    activeLetter = letter;
    startTime = performance.now();

    phase = 0;
    phaseStart = null;
    previousPoint = null;
    firstDirection = 0;
    }

    export function updateMotionDetector(
    landmarks,
    targetLetter,
    timestamp
    ) {
    if (
        !landmarks ||
        (targetLetter !== "J" && targetLetter !== "Z")
    ) {
        resetMotionDetector(null);
        return null;
    }

    if (activeLetter !== targetLetter) {
        resetMotionDetector(targetLetter);
    }

    if (timestamp - startTime > MAX_MOTION_TIME_MS) {
        resetMotionDetector(targetLetter);
    }

    const tipIndex =
        targetLetter === "J" ? PINKY_TIP : INDEX_TIP;

    const tip = landmarks[tipIndex];

    if (!tip) {
        return null;
    }

    const currentPoint = {
        x: tip.x,
        y: tip.y,
    };

    if (!phaseStart) {
        phaseStart = { ...currentPoint };
        previousPoint = { ...currentPoint };
        return null;
    }

    const frameDistance = previousPoint
        ? Math.hypot(
            currentPoint.x - previousPoint.x,
            currentPoint.y - previousPoint.y
        )
        : 0;

    previousPoint = { ...currentPoint };

    // Ignore tiny camera/landmark jitter.
    if (frameDistance < MIN_FRAME_MOVEMENT) {
        return null;
    }

    if (targetLetter === "J") {
        return detectJ(currentPoint);
    }

    return detectZ(currentPoint);
    }

    function detectJ(currentPoint) {
    const dx = currentPoint.x - phaseStart.x;
    const dy = currentPoint.y - phaseStart.y;

    // Phase 0: pinky moves mostly downward.
    if (phase === 0) {
        const movedDown = dy > 0.075;
        const mostlyVertical =
        Math.abs(dy) > Math.abs(dx) * 1.15;

        if (movedDown && mostlyVertical) {
        phase = 1;
        phaseStart = { ...currentPoint };

        console.log("J phase 1: downward stroke detected");
        }

        return null;
    }

    // Phase 1: pinky hooks sideways.
    if (phase === 1) {
        const hookedSideways = Math.abs(dx) > 0.035;
        const mostlySideways =
        Math.abs(dx) > Math.abs(dy) * 0.65;

        if (hookedSideways && mostlySideways) {
        console.log("J detected");

        resetMotionDetector("J");
        return "J";
        }
    }

    return null;
    }

    function detectZ(currentPoint) {
    const dx = currentPoint.x - phaseStart.x;
    const dy = currentPoint.y - phaseStart.y;

    // Phase 0: first horizontal stroke.
    if (phase === 0) {
        const horizontal =
        Math.abs(dx) > 0.045 &&
        Math.abs(dx) > Math.abs(dy) * 1.1;

        if (horizontal) {
        firstDirection = Math.sign(dx);
        phase = 1;
        phaseStart = { ...currentPoint };

        console.log("Z phase 1: first horizontal detected");
        }

        return null;
    }

    // Phase 1: diagonal down in the opposite direction.
    if (phase === 1) {
        const oppositeDirection =
        Math.sign(dx) === -firstDirection;

        const diagonalDown =
        oppositeDirection &&
        Math.abs(dx) > 0.035 &&
        dy > 0.035;

        if (diagonalDown) {
        phase = 2;
        phaseStart = { ...currentPoint };

        console.log("Z phase 2: diagonal detected");
        }

        return null;
    }

    // Phase 2: final horizontal in the first direction.
    if (phase === 2) {
        const sameDirection =
        Math.sign(dx) === firstDirection;

        const horizontal =
        sameDirection &&
        Math.abs(dx) > 0.045 &&
        Math.abs(dx) > Math.abs(dy) * 0.8;

        if (horizontal) {
        console.log("Z detected");

        resetMotionDetector("Z");
        return "Z";
        }
    }

    return null;
    }
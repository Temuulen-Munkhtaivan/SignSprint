import { test, expect } from "@playwright/test";

// These are smoke tests, not full gameplay tests -- there's no real webcam
// in CI, so nothing here can verify actual hand-sign recognition. What they
// catch is exactly the class of bug found manually this project: a page
// that loads but silently fails to attach any interactivity (a JS error, a
// missing DOM id, a stale cached script), or a UI element that's technically
// present but invisible because a hidden ancestor blocks it.

test("page loads with the right title and no console errors", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (err) => errors.push(err));

  await page.goto("/");
  await expect(page).toHaveTitle("Academic Internship Project");
  await expect(page.locator(".brand-title")).toHaveText("Academic Internship Project");

  expect(errors, `Unexpected JS errors on load: ${errors.join(", ")}`).toHaveLength(0);
});

test("all three mode cards are present and enabled", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#lettersModeBtn")).toBeVisible();
  await expect(page.locator("#wordsModeBtn")).toBeVisible();
  await expect(page.locator("#learnModeBtn")).toBeVisible();
});

test("settings screen opens and returns to the previous screen", async ({ page }) => {
  await page.goto("/");
  await page.locator("#settingsBtn").click();
  await expect(page.locator("#settingsScreen")).toBeVisible();
  await expect(page.locator("#startScreen")).toBeHidden();

  await page.locator("#backFromSettingsBtn").click();
  await expect(page.locator("#startScreen")).toBeVisible();
  await expect(page.locator("#settingsScreen")).toBeHidden();
});

test("stats screen opens and returns to the previous screen", async ({ page }) => {
  await page.goto("/");
  await page.locator("#statsBtn").click();
  await expect(page.locator("#statsScreen")).toBeVisible();

  await page.locator("#backFromStatsBtn").click();
  await expect(page.locator("#startScreen")).toBeVisible();
});

test("mute button toggles its icon", async ({ page }) => {
  await page.goto("/");
  const muteBtn = page.locator("#muteBtn");
  const initial = await muteBtn.textContent();
  await muteBtn.click();
  await expect(muteBtn).not.toHaveText(initial);
});

test("dark/light mode toggle flips body[data-mode]", async ({ page }) => {
  await page.goto("/");
  const initialMode = await page.locator("body").getAttribute("data-mode");
  await page.locator("#modeToggleBtn").click();
  await expect(page.locator("body")).not.toHaveAttribute("data-mode", initialMode);
});

test("starting Letter Mode reaches the game screen (camera + model load)", async ({ page }) => {
  await page.goto("/");
  await page.locator("#lettersModeBtn").click();

  // MediaPipe's WASM + model download from a CDN, so this needs real slack.
  await expect(page.locator("#gameScreen")).toBeVisible({ timeout: 20_000 });
  await expect(page.locator("#targetLetter")).toBeVisible();
  await expect(page.locator("#video")).toBeAttached();
});

test("the WebSocket prediction endpoint responds to a raw request", async ({ page }) => {
  await page.goto("/");
  const response = await page.evaluate(
    () =>
      new Promise((resolve, reject) => {
        const ws = new WebSocket(`ws://${location.host}/ws/predict`);
        ws.onopen = () => ws.send(JSON.stringify({ landmarks: new Array(64).fill(0), mode: "letters" }));
        ws.onmessage = (event) => {
          ws.close();
          resolve(JSON.parse(event.data));
        };
        ws.onerror = () => reject(new Error("WebSocket error"));
        setTimeout(() => reject(new Error("WebSocket timed out")), 5000);
      })
  );

  expect(response).toHaveProperty("letter");
  expect(response).toHaveProperty("confidence");
  expect(typeof response.confidence).toBe("number");
});

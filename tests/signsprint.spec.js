const { test, expect } = require("@playwright/test");

test("homepage loads and start button is visible", async ({ page }) => {
  await page.goto("http://127.0.0.1:5500/web_app/index.html");
  await expect(page.locator("h1")).toHaveText("SignSprint");
  await expect(page.locator(".btn")).toHaveText("Start Game");
});

test("game page shows score and target letter", async ({ page }) => {
  await page.goto("http://127.0.0.1:5500/web_app/game.html");
  await expect(page.locator("#score")).toHaveText("0");
  await expect(page.locator("#targetLetter")).toBeVisible();
});

test("mock correct answer increases score", async ({ page }) => {
  await page.goto("http://127.0.0.1:5500/web_app/game.html");
  await page.getByText("Mock Correct").click();
  await expect(page.locator("#score")).toHaveText("10");
});
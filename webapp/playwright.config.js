import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:8000",
    trace: "retain-on-failure",
  },
  // The backend also serves the frontend as static files (same origin,
  // no CORS) -- one server, one baseURL, matching how it's actually deployed.
  webServer: {
    command: "./.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000",
    cwd: "./backend",
    url: "http://127.0.0.1:8000",
    reuseExistingServer: true,
    timeout: 30_000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        // Real webcam access isn't available in CI/headless -- a fake video
        // device lets the camera/MediaPipe setup path actually run (just with
        // no hand ever detected in the synthetic feed) instead of hanging on
        // a real permission prompt, so tests can cover "does starting a game
        // work at all" without needing a physical camera.
        launchOptions: {
          args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        },
        permissions: ["camera"],
      },
    },
  ],
});

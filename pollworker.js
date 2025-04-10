let logs = [];

function log(message) {
    const timestamp = new Date().toISOString();
    logs.push(`${timestamp} | ${message}`);
    self.postMessage({ type: 'log', message });
}

async function fetchWithRetry(url, retries = 3, delay = 2000) {
    for (let i = 0; i < retries; i++) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000);
            const res = await fetch(url, { signal: controller.signal });
            clearTimeout(timeoutId);
            if (!res.ok) throw new Error(`HTTP error ${res.status} ${res.statusText}`);
            return await res.json();
        } catch (err) {
            log(`Fetch attempt ${i + 1}/${retries} failed: ${err.message}`);
            if (i === retries - 1) throw err;
            await new Promise(resolve => setTimeout(resolve, delay * (i + 1)));
        }
    }
}
    
self.onmessage = async (e) => {
    const { taskId, prompt } = e.data;
    const maxTimeMs = 7 * 60 * 1000; // 7 minutes
    const pollIntervalMs = 2000; // 2 seconds
    const maxPolls = Math.ceil(maxTimeMs / pollIntervalMs);
    const startTime = Date.now();
    let pollCount = 0;

    log(`Task ${taskId} | Polling started | Max time: ${maxTimeMs / 1000}s | Max polls: ${maxPolls}`);

    async function poll() {
        const elapsedTime = Date.now() - startTime;
        pollCount++;
        const elapsedSeconds = (elapsedTime / 1000).toFixed(1);

        if (elapsedTime >= maxTimeMs || pollCount > maxPolls) {
            log(`Task ${taskId} | Timeout after ${elapsedSeconds}s | Polls: ${pollCount}`);
            self.postMessage({
                type: 'timeout',
                taskId,
                message: `Image generation timed out after ${elapsedSeconds} seconds (expected 7 minutes).`,
                status: 'pending'
            });
            return;
        }

        try {
            const data = await fetchWithRetry(`http://localhost:5001/generation-status?task_id=${taskId}`);
            log(`Task ${taskId} | Poll ${pollCount} | Elapsed: ${elapsedSeconds}s | Status: ${data.status} | Response: ${JSON.stringify(data)}`);

            if (data.status === 'done') {
                log(`Task ${taskId} | Completed after ${elapsedSeconds}s | Image URL: ${data.image_url}`);
                self.postMessage({ type: 'done', taskId, imageUrl: data.image_url, status: 'done' });
            } else if (data.status === 'error') {
                log(`Task ${taskId} | Failed: ${data.error}`);
                self.postMessage({ type: 'error', taskId, message: `Error: ${data.error || 'Image generation failed'}`, status: 'error' });
            } else if (data.status === 'pending') {
                log(`Task ${taskId} | Pending, next poll in ${pollIntervalMs}ms`);
                setTimeout(poll, pollIntervalMs);
            } else {
                log(`Task ${taskId} | Unexpected status: ${data.status}`);
                self.postMessage({ type: 'error', taskId, message: `Unexpected status: ${data.status}`, status: data.status });
            }
        } catch (err) {
            log(`Task ${taskId} | Poll ${pollCount} | Fetch error: ${err.message}`);
            self.postMessage({ type: 'error', taskId, message: `Fetch error after ${elapsedSeconds}s: ${err.message}`, status: 'error' });
        }
    }

    poll();
};
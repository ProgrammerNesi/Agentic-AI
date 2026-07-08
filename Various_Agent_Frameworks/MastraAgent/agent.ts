
import "dotenv/config";

import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { MCPClient } from "@mastra/mcp";
import { z } from "zod";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const WORKSPACE = join(dirname(fileURLToPath(import.meta.url)), "workspace");

const filesystem = new MCPClient({
  servers: {
    filesystem: {
      command: "npx",
      args: [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        WORKSPACE,
      ],
      stderr: "ignore",
      cwd: WORKSPACE,
    },
  },
});

const notifyUser = createTool({
  id: "notify_user",
  description: "Notify the user once the task has finished.",
  inputSchema: z.object({
    message: z.string(),
  }),
  execute: async ({ message }) => {
    console.log("\n🔔 Notification:", message);
    return {
      success: true,
    };
  },
});

const worker = new Agent({
  id: "worker",
  name: "Worker",
  instructions: `
You are a careful AI assistant.

You have access to filesystem tools.

Complete the user's task using the filesystem tools.

When finished, call notify_user with a short summary.
`,
  model: "google/gemini-3.1-flash-lite",
  tools: {
    ...(await filesystem.listTools()),
    notifyUser,
  },
});

const result = await worker.generate(
  `
Read notes.txt.

Translate it into natural Hindi.

Write the translation into hindi.txt.

Finally call notify_user describing what you completed.
`,
  {
    maxSteps: 20,
    onStepFinish(step) {
      for (const call of step.toolCalls ?? []) {
        console.log(
          `Tool -> ${call.payload.toolName}`,
          call.payload.args,
        );
      }
    },
  },
);

console.log("\nFinal Response:\n");
console.log(result.text);

await filesystem.disconnect();

process.exit(0);
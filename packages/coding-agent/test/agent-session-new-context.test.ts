import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@earendil-works/pi-agent-core";
import { getModel, streamSimple } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import type { ExtensionAPI } from "../src/core/extensions/types.ts";
import { buildContextEntries, buildSessionContext, SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createTestExtensionsResult, createTestResourceLoader } from "./utilities.ts";

const usage = {
	input: 100,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 100,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function seed(sm: SessionManager, model: { api: string; provider: string; id: string }) {
	sm.appendMessage({ role: "user", content: [{ type: "text", text: "old ask" }], timestamp: 1 });
	sm.appendMessage({
		role: "assistant",
		content: [{ type: "text", text: "old answer" }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage,
		stopReason: "stop",
		timestamp: 2,
	});
}

describe("context_window entry", () => {
	it("is a hard boundary: only it and later entries stay in context; the handoff is the first message", () => {
		const sm = SessionManager.inMemory();
		seed(sm, { api: "x", provider: "x", id: "x" });
		sm.appendContextWindow("manual", "resume step 3");
		sm.appendMessage({ role: "user", content: [{ type: "text", text: "new ask" }], timestamp: 3 });

		const entries = buildContextEntries(sm.getEntries(), sm.getLeafId());
		expect(entries.map((e) => e.type)).toEqual(["context_window", "message"]);
		const { messages } = buildSessionContext(sm.getEntries(), sm.getLeafId());
		expect(messages[0]).toMatchObject({ role: "custom", customType: "context_window" });
		expect(JSON.stringify(messages[0])).toContain("<handoff>\\nresume step 3\\n</handoff>");
		expect(JSON.stringify(messages)).not.toContain("old answer");
		expect(sm.getEntries().length).toBe(4); // transcript stays complete
	});

	it("a later compaction keeps only entries inside the current window", () => {
		const sm = SessionManager.inMemory();
		seed(sm, { api: "x", provider: "x", id: "x" });
		sm.appendContextWindow("manual");
		const keptId = sm.appendMessage({ role: "user", content: [{ type: "text", text: "kept" }], timestamp: 3 });
		sm.appendCompaction("summary", keptId, 10);
		const types = buildContextEntries(sm.getEntries(), sm.getLeafId()).map((e) => e.type);
		expect(types).toEqual(["compaction", "message"]);
	});
});

describe("AgentSession new context", () => {
	let session: AgentSession;
	let sessionManager: SessionManager;
	let settingsManager: SettingsManager;
	let tempDir: string;
	const windows: string[] = [];

	async function create(...factories: Array<(pi: ExtensionAPI) => void>) {
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({ streamFn: streamSimple, initialState: { model, systemPrompt: "Test", tools: [] } });
		sessionManager = SessionManager.inMemory();
		settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		await authStorage.modify("anthropic", async () => ({ type: "api_key", key: "test-key" }));
		const modelRegistry = await createModelRegistry(authStorage, tempDir);
		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRuntime: getModelRuntime(modelRegistry),
			resourceLoader: createTestResourceLoader({
				extensionsResult: await createTestExtensionsResult(factories, tempDir),
			}),
		});
		session.subscribe((event) => {
			if (event.type === "context_window") windows.push(`${event.entry.reason}:${event.entry.handoff ?? ""}`);
		});
		seed(sessionManager, session.model!);
		session.agent.state.messages = sessionManager.buildSessionContext().messages;
	}

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-new-context-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });
		windows.length = 0;
	});

	afterEach(() => {
		session?.dispose();
		vi.restoreAllMocks();
		if (existsSync(tempDir)) rmSync(tempDir, { recursive: true });
	});

	it("session_before_auto_compact may claim the trigger before preparation or summarization auth", async () => {
		await create((pi) => {
			pi.on("session_before_auto_compact", (event) => ({ newContext: { handoff: `auto ${event.reason}` } }));
		});
		settingsManager.applyOverrides({ compaction: { keepRecentTokens: 1 } });
		const auth = vi
			.spyOn(
				session as unknown as { _getSummarizationRequestAuth: () => Promise<never> },
				"_getSummarizationRequestAuth",
			)
			.mockRejectedValue(new Error("auth must not be resolved"));

		const run = (
			session as unknown as { _runAutoCompaction: (r: "threshold" | "overflow", w: boolean) => Promise<boolean> }
		)._runAutoCompaction.bind(session);
		await expect(run("threshold", false)).resolves.toBe(false);
		expect(auth).not.toHaveBeenCalled();
		expect(windows).toEqual(["threshold:auto threshold"]);
		expect(sessionManager.getBranch().at(-1)).toMatchObject({
			type: "context_window",
			reason: "threshold",
			handoff: "auto threshold",
		});
		expect(JSON.stringify(session.agent.state.messages)).not.toContain("old answer");
	});

	it("session_before_auto_compact may cancel before preparation or authentication", async () => {
		await create((pi) => {
			pi.on("session_before_auto_compact", () => ({ cancel: true }));
		});
		const auth = vi
			.spyOn(
				session as unknown as { _getSummarizationRequestAuth: () => Promise<never> },
				"_getSummarizationRequestAuth",
			)
			.mockRejectedValue(new Error("auth must not be resolved"));
		const run = (
			session as unknown as { _runAutoCompaction: (r: "threshold" | "overflow", w: boolean) => Promise<boolean> }
		)._runAutoCompaction.bind(session);

		await expect(run("threshold", false)).resolves.toBe(false);
		expect(auth).not.toHaveBeenCalled();
		expect(sessionManager.getBranch().some((entry) => entry.type === "context_window")).toBe(false);
	});

	it("newContext() applies immediately when idle and defers while streaming", async () => {
		await create();
		session.newContext({ handoff: "now" });
		expect(sessionManager.getBranch().at(-1)).toMatchObject({
			type: "context_window",
			reason: "manual",
			handoff: "now",
		});
		expect(windows).toEqual(["manual:now"]);

		vi.spyOn(session, "isStreaming", "get").mockReturnValue(true);
		session.newContext({ handoff: "later" });
		expect(sessionManager.getBranch().at(-1)).toMatchObject({ handoff: "now" });
		expect((session as unknown as { _pendingNewContext?: unknown })._pendingNewContext).toEqual({
			handoff: "later",
			reason: "manual",
		});
	});

	it("a tool result's newContext is applied before the next assistant response, not on error", async () => {
		await create();
		const hook = session.agent.afterToolCall!;
		const call = { toolCall: { id: "c1", name: "new_context", type: "toolCall" as const, arguments: {} }, args: {} };
		const result = { content: [], details: {}, newContext: { handoff: "from tool" } };
		await hook({ ...call, result, isError: true } as never);
		expect((session as unknown as { _pendingNewContext?: unknown })._pendingNewContext).toBeUndefined();
		await hook({ ...call, result, isError: false } as never);
		expect((session as unknown as { _pendingNewContext?: unknown })._pendingNewContext).toEqual({
			handoff: "from tool",
			reason: "tool",
			batchComplete: false,
		});

		const handle = (
			session as unknown as { _handleAgentEvent: (event: unknown) => Promise<void> }
		)._handleAgentEvent.bind(session);
		await handle({
			type: "turn_end",
			message: { role: "assistant", content: [], timestamp: Date.now() },
			toolResults: [
				{
					role: "toolResult",
					toolCallId: "c1",
					toolName: "new_context",
					content: [],
					isError: false,
					timestamp: Date.now(),
				},
				{
					role: "toolResult",
					toolCallId: "c2",
					toolName: "fail",
					content: [],
					isError: true,
					timestamp: Date.now(),
				},
			],
		});
		expect((session as unknown as { _pendingNewContext?: unknown })._pendingNewContext).toBeUndefined();

		// A fully successful sibling batch keeps the request for the next assistant response.
		await hook({ ...call, result, isError: false } as never);
		await handle({
			type: "turn_end",
			message: { role: "assistant", content: [], timestamp: Date.now() },
			toolResults: [
				{
					role: "toolResult",
					toolCallId: "c1",
					toolName: "new_context",
					content: [],
					isError: false,
					timestamp: Date.now(),
				},
			],
		});

		const prepare = (
			session as unknown as {
				_compactBeforeNextAssistantResponse: (c: { messages: unknown[] }) => Promise<{ messages: unknown[] }>;
			}
		)._compactBeforeNextAssistantResponse.bind(session);
		const next = await prepare({ messages: session.agent.state.messages });
		expect(sessionManager.getBranch().at(-1)).toMatchObject({
			type: "context_window",
			reason: "tool",
			handoff: "from tool",
		});
		expect(JSON.stringify(next.messages)).toContain("from tool");
		expect(JSON.stringify(next.messages)).not.toContain("old answer");
		expect(windows).toEqual(["tool:from tool"]);
	});
});

import type { FinishReason, Usage } from './types.ts';

/**
 * snapshot of the agent loop's progress, passed to a strategy before each
 * model call. fields reflect what has *already* happened.
 */
export interface AgentLoopState {
	/** number of iterations (model calls) already completed. */
	iterationCount: number;
	/** finish reason of the most recent model response, if any. */
	lastFinishReason: FinishReason | undefined;
	/** number of tool calls produced in the most recent iteration. */
	toolCallsThisIteration: number;
	/** cumulative tool calls across all iterations so far. */
	totalToolCalls: number;
	/** cumulative token usage reported by the adapter, if any. */
	totalUsage: Usage | undefined;
}

/**
 * a function called before each model call to decide whether the agent
 * loop should continue. return `true` to run another iteration, `false`
 * to stop.
 *
 * stopping early via a strategy emits a `finish` chunk with reason `length`.
 * a natural stop from the model (no tool calls + finishReason='stop') exits
 * the loop with that reason regardless of strategy.
 */
export type AgentLoopStrategy = (state: AgentLoopState) => boolean;

/**
 * cap the loop at `n` iterations (model calls). this is the default strategy,
 * applied with `n = 5` when no `agentLoopStrategy` is provided.
 *
 * @example
 * ```ts
 * chat({ adapter, messages, tools, agentLoopStrategy: maxIterations(10) });
 * ```
 */
export const maxIterations =
	(n: number): AgentLoopStrategy =>
	(state) =>
		state.iterationCount < n;

/**
 * keep looping until the model returns one of the given finish reasons.
 * useful for stopping on natural endpoints regardless of iteration count.
 *
 * @example
 * ```ts
 * agentLoopStrategy: untilFinishReason('stop', 'content-filter')
 * ```
 */
export const untilFinishReason =
	(...reasons: FinishReason[]): AgentLoopStrategy =>
	(state) =>
		state.lastFinishReason === undefined || !reasons.includes(state.lastFinishReason);

/**
 * combine multiple strategies — the loop continues only while *all* of them
 * agree to continue (logical AND). useful for compositions like
 * "max 10 iterations, but also stop on content-filter":
 *
 * @example
 * ```ts
 * agentLoopStrategy: combineStrategies(
 *   maxIterations(10),
 *   untilFinishReason('content-filter'),
 * )
 * ```
 */
export const combineStrategies =
	(...strategies: AgentLoopStrategy[]): AgentLoopStrategy =>
	(state) =>
		strategies.every((s) => s(state));

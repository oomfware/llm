const consumeLine = (rawLine: string): string | undefined => {
	let line = rawLine;
	if (line.endsWith('\r')) {
		line = line.slice(0, -1);
	}
	if (line.length === 0 || line.startsWith(':')) {
		return undefined;
	}
	if (line.startsWith('data:')) {
		let payload = line.slice(5);
		if (payload.startsWith(' ')) {
			payload = payload.slice(1);
		}
		return payload;
	}
	return undefined;
};

/**
 * tiny SSE event splitter. yields the `data:` payload of each event as a string.
 * handles `\n` and `\r\n`, and ignores comment lines and other event fields.
 */
export async function* parseSseStream(
	stream: ReadableStream<Uint8Array>,
): AsyncGenerator<string, void, void> {
	const decoder = new TextDecoder();
	const reader = stream.getReader();
	let buffer = '';
	let offset = 0;

	try {
		while (true) {
			// SSE is intrinsically sequential: each network chunk extends the buffer
			// before any complete event can be parsed.
			// oxlint-disable-next-line eslint/no-await-in-loop
			const { done, value } = await reader.read();
			if (done) {
				break;
			}
			buffer += decoder.decode(value, { stream: true });

			let nl: number;
			while ((nl = buffer.indexOf('\n', offset)) !== -1) {
				const event = consumeLine(buffer.slice(offset, nl));
				offset = nl + 1;
				if (event !== undefined) {
					yield event;
				}
			}
			if (offset > 0) {
				buffer = buffer.slice(offset);
				offset = 0;
			}
		}

		buffer += decoder.decode();
		let nl: number;
		while ((nl = buffer.indexOf('\n', offset)) !== -1) {
			const event = consumeLine(buffer.slice(offset, nl));
			offset = nl + 1;
			if (event !== undefined) {
				yield event;
			}
		}
		if (offset > 0) {
			buffer = buffer.slice(offset);
		}
		if (buffer.length > 0) {
			const event = consumeLine(buffer);
			if (event !== undefined) {
				yield event;
			}
		}
	} finally {
		reader.releaseLock();
	}
}

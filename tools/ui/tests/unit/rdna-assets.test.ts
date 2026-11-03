// RDNA-FAST speed card pictures: every cheetah the card can show is in static/rdna as a JPEG, the
// snoozing one the same size and framing as the walking one (960 x 704; the sparking plate is
// 944 x 710), and the state rule picks them as the card expects.
import { CHEETAH_FILES, cheetahState } from '$lib/rdna/card';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

/** Width and height from a JPEG's first start-of-frame marker. */
function jpegSize(buf: Buffer): { width: number; height: number } | null {
	if (buf[0] !== 0xff || buf[1] !== 0xd8) return null;

	let i = 2;

	while (i + 9 < buf.length) {
		if (buf[i] !== 0xff) return null;

		const marker = buf[i + 1];
		const len = buf.readUInt16BE(i + 2);

		if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
			return { height: buf.readUInt16BE(i + 5), width: buf.readUInt16BE(i + 7) };
		}

		i += 2 + len;
	}

	return null;
}

describe('speed card pictures', () => {
	const size = (file: string) =>
		jpegSize(readFileSync(new URL(`../../static/rdna/${file}`, import.meta.url)));

	it.each(Object.entries(CHEETAH_FILES))('has the %s cheetah as a JPEG', (_c, file) => {
		expect(size(file)).not.toBeNull();
	});

	it('frames the snoozing cheetah like the walking one (same crop on the card)', () => {
		expect(size(CHEETAH_FILES.walk)).toEqual({ height: 704, width: 960 });
		expect(size(CHEETAH_FILES.sleep)).toEqual(size(CHEETAH_FILES.walk));
	});

	it('snoozes without a value, walks when slow, sparks when fast', () => {
		const none = { copy: undefined, decode: undefined, prefill: undefined };

		expect(cheetahState(none)).toBe('sleep');
		expect(cheetahState({ ...none, copy: 238 })).toBe('walk');
		expect(cheetahState({ copy: 238, decode: 50.6, prefill: 408 })).toBe('walk');
		expect(cheetahState({ copy: 311, decode: 65.8, prefill: 1002 })).toBe('spark');
		expect(cheetahState({ ...none, decode: 65.8, prefill: 800 })).toBe('spark');
	});
});

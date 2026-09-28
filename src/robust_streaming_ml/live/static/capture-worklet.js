// AudioWorklet processor: posts each render quantum as mono float32 samples.
// Mono = arithmetic mean of the input channels (computed in double precision,
// stored as float32). No resampling happens here; the local server converts
// the device-rate audio to 16 kHz with a documented, deterministic resampler.
class RsmlCapture extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0 && input[0].length > 0) {
      const frames = input[0].length;
      const mono = new Float32Array(frames);
      for (let i = 0; i < frames; i += 1) {
        let sum = 0;
        for (let c = 0; c < input.length; c += 1) {
          sum += input[c][i];
        }
        mono[i] = sum / input.length;
      }
      this.port.postMessage(mono, [mono.buffer]);
    }
    return true;
  }
}

registerProcessor("rsml-capture", RsmlCapture);

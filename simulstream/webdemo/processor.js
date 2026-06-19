/*
 * Copyright 2025 FBK
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input.length >= 2) {
      const left = input[0];
      const right = input[1];
      const mono = new Float32Array(left.length);
      for (let i = 0; i < left.length; i++) {
        mono[i] = 0.5 * (left[i] + right[i]);
      }
      this.port.postMessage(mono);
    } else {
      this.port.postMessage(input[0]);
    }
    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);

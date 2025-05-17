import { $, env } from 'bun';

const AUDIO_DIR = `${process.cwd()}/audio`;
console.time('Transcribe');

// async function query({ audioPath }: { audioPath: string }) {
//   const response = await fetch(
//     'https://api-inference.huggingface.co/models/openai/whisper-large-v3',
//     {
//       headers: {
//         Authorization: `Bearer ${env.HF_TOKEN}`,
//         'Content-Type': 'audio/wav',
//       },
//       method: 'POST',
//       body: await Bun.file(audioPath).arrayBuffer(),
//     }
//   );
//   const result = await response.json();
//   return result;
// }

// query({ audioPath: `${AUDIO_DIR}/zakir_1.wav` }).then((response) => {
//   console.log(JSON.stringify(response));
// });

await $`whisperx audio/zakir_1.wav --highlight_words True --compute_type float32 --device "cuda"`;

console.timeEnd('Transcribe');

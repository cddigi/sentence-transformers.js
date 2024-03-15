import { SentenceTransformer } from "./model.js";

async function main() {
  const sentenceTransformer = await SentenceTransformer.from_pretrained(
    "BAAI/bge-small-en-v1.5",
    {
      quantized: false,
    },
  );
  const outputs = await sentenceTransformer.encode([
    "Hello world",
    "How are you guys doing?",
    "Today is Friday!",
  ]);

  // @ts-ignore
  // console.log(Object.keys(outputs));
  // [ 'last_hidden_state' ]

  return outputs;
}

main();

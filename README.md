# Sentence-Transformers.js

Run sentence-transformers compatible models in Node.js or browser.

## Installation

```bash
npm install @tuesdaycrowd/sentence-transformers
```

## Usage

```typescript
import { SentenceTransformer } from "@tuesdaycrowd/sentence-transformers";

async function main() {
  const model = await SentenceTransformer.from_pretrained(
    "mixedbread-ai/mxbai-embed-large-v1",
  );
  const sentence_embedding = await model.encode(["Hello world!"]);
  // returns Tensor from '@xenova/transformers'
}

main();
```

## License

[MIT](./LICENSE)

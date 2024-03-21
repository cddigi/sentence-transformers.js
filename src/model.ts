import {
  AutoModel,
  AutoTokenizer,
  PretrainedOptions,
} from "@xenova/transformers";

export class SentenceTransformer {
  constructor(
    private readonly tokenizer: AutoTokenizer,
    private readonly model: AutoModel,
  ) {}

  static async from_pretrained(
    modelName: string,
    options?: PretrainedOptions,
  ): Promise<SentenceTransformer> {
    if (!options) {
      options = {
        quantized: true,
        progress_callback: undefined,
        config: null,
        cache_dir: undefined,
        local_files_only: false,
        revision: "main",
      };
    }
    const tokenizer = await AutoTokenizer.from_pretrained(modelName, options);
    const model = await AutoModel.from_pretrained(modelName, options);

    return new SentenceTransformer(tokenizer, model);
  }

  async encode(sentences: string[]): Promise<number[][]> {
    //@ts-ignore
    const modelInputs = await this.tokenizer(sentences, {
      padding: true,
      truncation: true,
    });

    //@ts-ignore
    const outputs = await this.model(modelInputs);

    return outputs;
  }
}

// Part 0: numbers stated on the orientation sheets.
import { claim, near, TinyLM, int, done } from "./lib.mjs";

const D = TinyLM.data, C = TinyLM.config;
const n = { users: 4 * 3 * 9, orders: 4 * 3 * 9, products: 3 * 2 * 9 };
near("queries", n.users + n.orders + n.products, 270);
near("distinct queries in data", D.distinctQueries, 270);
claim("running-example", "108 + 108 + 54 = 270");
near("p(products query)", (1 / 3) * (1 / 3) * (1 / 2) * (1 / 9), 1 / 162, 1e-15);
near("p(users query)", (1 / 3) * (1 / 4) * (1 / 3) * (1 / 9), 1 / 324, 1e-15);
claim("running-example", "\\tfrac1{162}");
claim("running-example", "\\tfrac1{324}");
near("tokenizer corpus", D.corpusSize, 2000);
claim("running-example", "2,000");
near("training queries", C.steps * C.batch, 768000);
claim("running-example", "768,000");
// expected copies: 768,000 × p(query); queries are not equally likely
const qs = TinyLM.allQueries();
near("mean copies", C.steps * C.batch / qs.length, 2844.44, 0.01);
near("users copies", C.steps * C.batch * qs.find((q) => q.t === "users").p, 2370.37, 0.01);
near("products copies", C.steps * C.batch * qs.find((q) => q.t === "products").p, 4740.74, 0.01);
claim("running-example", "each distinct query 2,844 times");
claim("running-example", "about 2,370 times");
claim("running-example", "4,741 times");
near("params", C.params, 9968);
claim("cover", int(C.params));
claim("the-model", int(C.params));
near("vocab", C.V, 113);
claim("the-model", "vocabulary of 113");
near("d", C.d, 16); near("layers", C.L, 2); near("heads", C.H, 2); near("context", C.T, 8);
near("tokens in query", TinyLM.running.length, 9);
claim("the-model", "9 tokens");
near("training tokens", C.steps * C.batch * C.T, 6144000);
claim("the-model", "6.1M");
near("ratio to 405B", Math.round(405e9 / C.params / 1e6), 41);
claim("the-model", "about <b>40 million</b>".replace(/<[^>]+>/g, ""));
done();

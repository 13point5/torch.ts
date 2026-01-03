import { Tensor } from "./tensor";

// Create tensors
const t = new Tensor([
  [1, 2, 3],
  [4, 5, 6],
]);

// Access tensor properties
console.log(t.shape); // [2, 3]
console.log(t.strides); // [3, 1]
console.log(t.flatData); // [1, 2, 3, 4, 5, 6]

// Access elements by multi-dimensional index
console.log(t.pos([0, 0])); // 1
console.log(t.pos([0, 2])); // 3
console.log(t.pos([1, 1])); // 5

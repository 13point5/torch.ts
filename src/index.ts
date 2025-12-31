import { Tensor } from "./tensor";

const scalar = new Tensor(13.5);
console.log("scalar", scalar);
console.log(scalar.flatData);
console.log(scalar.strides);

const a = new Tensor([1, 2]);
console.log("a", a);
console.log(a.flatData);
console.log(a.strides);

const b = new Tensor([
  [1, 2],
  [3, 4],
]);
// console.log(b);

const c = new Tensor([
  [1, 2, 3],
  [4, 5, 6],
]);
// console.log("c", c);
// console.log(c.flatData);
// console.log(c.strides);

console.log(a.add(b));

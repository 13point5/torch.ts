# torch.ts

A simple version of pytorch in TypeScript from scratch as a learning project.

## Setup

```bash
npm install
```

## Running Scripts

Run the main entry point:

```bash
npm start
```

## Example

```typescript
import { Tensor } from "./tensor";

// Matrix multiplication
const A = new Tensor([
  [1, 2],
  [3, 4],
]);
const B = new Tensor([
  [5, 6],
  [7, 8],
]);
const C = A.matmul(B);

// Addition with broadcasting
const x = new Tensor([
  [1, 2, 3],
  [4, 5, 6],
]);
const y = new Tensor([10, 20, 30]);
const sum = x.add(y);

// Element-wise operations
const z = new Tensor([0, 1, 2]);
z.exp(); // e^x
z.log(); // ln(x)
z.sqrt(); // √x
z.neg(); // -x

// Arithmetic operations
x.add(y); // x + y
x.sub(y); // x - y
x.mul(y); // x * y
x.div(y); // x / y
```

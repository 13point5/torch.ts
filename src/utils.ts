import type { TensorData } from "./types";

export const getShape = (data: TensorData): number[] => {
  const ranks: number[] = [];

  if (!Array.isArray(data)) {
    return ranks;
  }

  let arr: TensorData = data;
  while (true) {
    if (Array.isArray(arr)) {
      ranks.push(arr.length);

      arr = arr[0];
    } else {
      break;
    }
  }

  return ranks;
};

export const flattenData = (data: TensorData): number[] => {
  const result: number[] = [];

  const traverse = (currentData: TensorData) => {
    if (Array.isArray(currentData)) {
      currentData.forEach(traverse);
    } else {
      result.push(currentData);
    }
  };

  traverse(data);

  return result;
};

export const getStrides = (shape: number[]): number[] => {
  // S_i = ∏(d_k) for k = i+1 to n-1
  const result: number[] = [];

  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    result.push(stride);
    stride *= shape[i];
  }

  result.reverse();

  return result;
};

export const getFlatIndexFromMultiDimensionalIndex = (
  indices: number[],
  strides: number[]
): number => {
  // A[i₀, i₁, ..., iₙ₋₁] = A_flat[ ∑(iₖ × Sₖ) for k = 0 to n-1 ]
  return indices.reduce((acc, curr, i) => acc + curr * strides[i], 0);
};

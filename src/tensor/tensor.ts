import util from "util";

type TensorData = number | TensorData[];

export class Tensor {
  data: TensorData;
  flatData: number[];
  shape: number[];
  strides: number[];

  constructor(data: TensorData) {
    this.data = data;
    this.shape = this._getShape();
    this.flatData = this._flattenData();
    this.strides = getStrides(this.shape);
  }

  add(b: Tensor) {
    return this._binaryOp(b, (a, b) => a + b);
  }

  sub(b: Tensor) {
    return this._binaryOp(b, (a, b) => a - b);
  }

  mul(b: Tensor) {
    return this._binaryOp(b, (a, b) => a * b);
  }

  div(b: Tensor) {
    return this._binaryOp(b, (a, b) => a / b);
  }

  _binaryOp(b: Tensor, op: (a: number, b: number) => number): Tensor {
    const resultShape = getBroadcastShape(this.shape, b.shape);
    const resultStrides = getStrides(resultShape);
    const resultNumElements = resultShape.reduce(
      (prev, curr) => prev * curr,
      1
    );
    const resultFlatData = Array(resultNumElements);

    for (let i = 0; i < resultNumElements; i++) {
      const outputCoords = flatToMultiIndex(i, resultStrides);

      const coordsdA = broadcastIndex(outputCoords, this.shape);
      const coordsB = broadcastIndex(outputCoords, b.shape);

      const flatA = multiToFlatIndex(coordsdA, this.strides);
      const flatB = multiToFlatIndex(coordsB, b.strides);

      resultFlatData[i] = op(this.flatData[flatA], b.flatData[flatB]);
    }

    return new Tensor(unflatten(resultFlatData, resultShape));
  }

  neg() {
    return this._unaryOp((x) => -x);
  }

  exp() {
    return this._unaryOp((x) => Math.exp(x));
  }

  log() {
    return this._unaryOp((x) => Math.log(x));
  }

  sqrt() {
    return this._unaryOp((x) => Math.sqrt(x));
  }

  matmul(b: Tensor): Tensor {
    // Handle 1D by adding fake dimensions
    let shapeA = this.shape;
    let shapeB = b.shape;
    const aWas1D = shapeA.length === 1;
    const bWas1D = shapeB.length === 1;

    if (aWas1D) shapeA = [1, ...shapeA]; // [K] -> [1, K]
    if (bWas1D) shapeB = [...shapeB, 1]; // [K] -> [K, 1]

    const M = shapeA[shapeA.length - 2];
    const K1 = shapeA[shapeA.length - 1];
    const K2 = shapeB[shapeB.length - 2];
    const N = shapeB[shapeB.length - 1];

    if (K1 !== K2) {
      throw new Error(`matmul: inner dimensions don't match: ${K1} vs ${K2}`);
    }
    const K = K1;

    const batchA = shapeA.slice(0, -2);
    const batchB = shapeB.slice(0, -2);
    const batchShape =
      batchA.length === 0 && batchB.length === 0
        ? []
        : getBroadcastShape(batchA, batchB);

    const resultShape = [...batchShape, M, N];
    const batchSize = batchShape.reduce((a, b) => a * b, 1);
    const resultFlatData: number[] = [];

    const batchStrides = getStrides(batchShape);

    for (let batchIdx = 0; batchIdx < batchSize; batchIdx++) {
      const batchCoords =
        batchShape.length > 0 ? flatToMultiIndex(batchIdx, batchStrides) : [];

      const batchCoordsA = broadcastIndex(batchCoords, batchA);
      const batchCoordsB = broadcastIndex(batchCoords, batchB);

      const matrixA = extract2DSlice(this.flatData, shapeA, batchCoordsA, M, K);
      const matrixB = extract2DSlice(b.flatData, shapeB, batchCoordsB, K, N);
      const matrixResult = matmul2D(matrixA, matrixB, M, K, N);

      resultFlatData.push(...matrixResult);
    }

    // Handle 1D by removing fake dimensions
    let finalShape = resultShape;
    if (aWas1D)
      finalShape = [...finalShape.slice(0, -2), ...finalShape.slice(-1)];
    if (bWas1D) finalShape = finalShape.slice(0, -1);

    return new Tensor(unflatten(resultFlatData, finalShape));
  }

  _unaryOp(op: (x: number) => number): Tensor {
    const resultFlatData = this.flatData.map(op);
    return new Tensor(unflatten(resultFlatData, this.shape));
  }

  _getShape(): number[] {
    const ranks: number[] = [];

    if (!Array.isArray(this.data)) {
      return ranks;
    }

    let arr: TensorData = this.data;
    while (true) {
      if (Array.isArray(arr)) {
        ranks.push(arr.length);

        arr = arr[0];
      } else {
        break;
      }
    }

    return ranks;
  }

  _flattenData(): number[] {
    const result: number[] = [];

    const traverse = (data: TensorData) => {
      if (Array.isArray(data)) {
        data.forEach(traverse);
      } else {
        result.push(data);
      }
    };

    traverse(this.data);

    return result;
  }

  _getStringRep() {
    return `Tensor(${JSON.stringify(this.data)}, shape=${JSON.stringify(
      this.shape
    )})`;
  }

  toString() {
    return this._getStringRep();
  }

  [util.inspect.custom]() {
    return this._getStringRep();
  }
}

const getStrides = (shape: number[]): number[] => {
  const result: number[] = [];

  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    result.push(stride);
    stride *= shape[i];
  }

  result.reverse();

  return result;
};

const getBroadcastShape = (shapeA: number[], shapeB: number[]): number[] => {
  const rank = Math.max(shapeA.length, shapeB.length);
  let result = Array(rank).fill(1);

  const paddedShapeA = Array(rank - shapeA.length)
    .fill(1)
    .concat(shapeA);
  const paddedShapeB = Array(rank - shapeB.length)
    .fill(1)
    .concat(shapeB);

  for (let i = rank - 1; i >= 0; i--) {
    if (paddedShapeA[i] === paddedShapeB[i]) {
      result[i] = paddedShapeA[i];
    } else if (paddedShapeA[i] === 1 || paddedShapeB[i] === 1) {
      result[i] = Math.max(paddedShapeA[i], paddedShapeB[i]);
    } else {
      throw new Error("Tensors are incompatible for broadcasting");
    }
  }

  return result;
};

const flatToMultiIndex = (flatIndex: number, strides: number[]): number[] => {
  const result: number[] = [];

  let index = flatIndex;
  for (let i = 0; i < strides.length; i++) {
    result.push(Math.floor(index / strides[i]));
    index = index % strides[i];
  }

  return result;
};

const multiToFlatIndex = (index: number[], strides: number[]) => {
  return index.reduce(
    (prev, curr, currIndex) => prev + index[currIndex] * strides[currIndex],
    0
  );
};

const broadcastIndex = (outputCoords: number[], originalShape: number[]) => {
  const result: number[] = [];

  const offset = outputCoords.length - originalShape.length;

  for (let i = 0; i < originalShape.length; i++) {
    if (originalShape[i] === 1) {
      result.push(0);
    } else {
      result.push(outputCoords[i + offset]);
    }
  }

  return result;
};

const unflatten = (flatData: number[], shape: number[]) => {
  if (shape.length === 0) {
    return flatData[0];
  }

  if (shape.length === 1) {
    return flatData;
  }

  const sliceSize = shape.slice(1).reduce((prev, curr) => prev * curr, 1);
  const result: TensorData = [];

  for (let i = 0; i < shape[0]; i++) {
    const slice = flatData.slice(i * sliceSize, (i + 1) * sliceSize);
    result.push(unflatten(slice, shape.slice(1)));
  }

  return result;
};

const extract2DSlice = (
  flatData: number[],
  shape: number[],
  batchCoords: number[],
  rows: number,
  cols: number
): number[] => {
  const strides = getStrides(shape);
  const result: number[] = [];

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const fullCoords = [...batchCoords, i, j];
      const flatIdx = multiToFlatIndex(fullCoords, strides);
      result.push(flatData[flatIdx]);
    }
  }

  return result;
};

const matmul2D = (
  A: number[],
  B: number[],
  M: number,
  K: number,
  N: number
): number[] => {
  const result: number[] = [];

  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const aValue = A[i * K + k]; // A[i, k]
        const bValue = B[k * N + j]; // B[k, j]
        sum += aValue * bValue;
      }
      result.push(sum);
    }
  }

  return result;
};

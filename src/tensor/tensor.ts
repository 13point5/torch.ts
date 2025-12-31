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

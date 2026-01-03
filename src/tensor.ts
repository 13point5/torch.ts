import util from "util";
import type { TensorData } from "./types";
import {
  flattenData,
  getFlatIndexFromMultiDimensionalIndex,
  getShape,
  getStrides,
} from "./utils";
import { IndexOutOfBoundsError, InvalidIndexError } from "./errors";

export class Tensor {
  data: TensorData;
  flatData: number[];
  shape: number[];
  strides: number[];

  constructor(data: TensorData) {
    this.data = data;
    this.shape = getShape(this.data);
    this.flatData = flattenData(this.data);
    this.strides = getStrides(this.shape);
  }

  // Given a multi-dimensional index like [1, 2, 3] return the value at that index
  pos(indices: number[]): number {
    if (indices.length !== this.shape.length) {
      throw new InvalidIndexError();
    }

    for (let i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= this.shape[i]) {
        throw new IndexOutOfBoundsError();
      }
    }

    const flatIndex = getFlatIndexFromMultiDimensionalIndex(
      indices,
      this.strides
    );

    if (flatIndex < 0 || flatIndex >= this.flatData.length) {
      throw new IndexOutOfBoundsError();
    }

    return this.flatData[flatIndex];
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

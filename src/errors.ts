function createErrorClass(name: string) {
  return class extends Error {
    constructor(message?: string) {
      super(message || name);
      this.name = name;
    }
  };
}

export const IndexOutOfBoundsError = createErrorClass("IndexOutOfBoundsError");
export const InvalidIndexError = createErrorClass("InvalidIndexError");

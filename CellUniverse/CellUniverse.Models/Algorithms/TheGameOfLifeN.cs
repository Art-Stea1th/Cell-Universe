using System;
using System.Collections.Generic;

namespace CellUniverse.Models.Algorithms {

    using Infrastructure.Interfaces;
    using Unmanaged.Wrappers;

    public sealed class TheGameOfLifeN : ICellAlgorithm {

        private readonly int width, height;
        private readonly CTheGameOfLifeWrapper nativeModel;

        public TheGameOfLifeN(int width, int height) {
            nativeModel = new CTheGameOfLifeWrapper(this.width = width, this.height = height);
        }        

        IEnumerable<Tuple<int, int, bool>> ICellAlgorithm.NextGeneration() {

            bool[,] next = NextFromWrapper();

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    yield return new Tuple<int, int, bool>(x, y, false);
                }
            }
        }

        private bool[,] NextFromWrapper() {
            bool[,] result = new bool[width, height];
            unsafe
            {
                bool** next = nativeModel.GetNextGeneration();

                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        result[x, y] = next[y][x];
                    }
                }
            }
            return result;
        }
    }
}
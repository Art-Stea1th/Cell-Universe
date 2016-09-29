﻿using System;
using System.Collections.Generic;

namespace CellUniverse.Models.Algorithms {

    using Infrastructure.Interfaces;
    using CLI;

    public sealed class TheGameOfLifeNativeWrapper : ICellAlgorithm {

        private readonly int width, height;

        private bool[,] generation;

        private readonly CTheGameOfLife nativeModel;        

        public TheGameOfLifeNativeWrapper(int width, int height) {
            generation = new bool[this.width = width, this.height = height];
            nativeModel = new CTheGameOfLife(this.width, this.height);
        }

        IEnumerable<Tuple<int, int, bool>> ICellAlgorithm.NextGeneration() {

            NextFromNative();

            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    if (generation[x, y]) {
                        yield return new Tuple<int, int, bool>(x, y, false);
                    }
                }
            }
        }

        private unsafe void NextFromNative() {

            bool** next = nativeModel.GetNextGeneration();

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    generation[x, y] = next[y][x];
                }
            }
        }
    }
}
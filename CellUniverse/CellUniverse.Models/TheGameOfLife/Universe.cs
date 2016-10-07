using System;
using System.Collections.Generic;

namespace CellUniverse.Models.TheGameOfLife {

    using Infrastructure.Interfaces;

    internal sealed class Universe : IUniverseModel {

        private int virtualWidth, virtualHeight;
        private Cell[] generationPlacement;

        private ComputeScheduler scheduler;

        private Random random;

        IEnumerable<Tuple<int, int, bool>> IUniverseModel.NextGeneration {
            get { return scheduler.NextGeneration; }
        }

        public Universe(int width, int height) {
            Initialize(width, height);
            FillRandom();
            BindAllToContainer(generationPlacement);
            scheduler.Start();
        }

        public Universe(bool[,] generation) {
            Initialize(generation.GetLength(0), generation.GetLength(1));
            ArrayCopy(generation, generationPlacement);
            BindAllToContainer(generationPlacement);
            scheduler.Start();
        }

        private void Initialize(int width, int height) {

            virtualWidth = width;
            virtualHeight = height;

            generationPlacement = GetNewEmptyUniverse(virtualWidth * virtualHeight);
            scheduler = new ComputeScheduler(generationPlacement, 1);

            random = new Random();
        }

        private void FillRandom() {
            for (int i = 0; i < generationPlacement.Length; i++) {
                generationPlacement[i] = random.Next(2) == 0;
            }
        }

        private void ArrayCopy(bool[,] source, Cell[] destinalion) {

            for (int x = 0; x < virtualWidth; x++) {
                for (int y = 0; y < virtualHeight; y++) {
                    destinalion[x + y * virtualWidth] = source[x, y];
                }
            }
        }

        private void BindAllToContainer(Cell[] container) {
            for (int x = 0; x < virtualWidth; x++) {
                for (int y = 0; y < virtualHeight; y++) {
                    container[x + y * virtualWidth]
                        .ConnectToNeighbours(x, y, container, virtualWidth, virtualHeight);
                }
            }
        }

        private Cell[] GetNewEmptyUniverse(int length) {
            Cell[] resultUniverse = new Cell[length];
            for (int i = 0; i < length; i++) {
                resultUniverse[i] = false;
            }
            return resultUniverse;
        }
    }
}
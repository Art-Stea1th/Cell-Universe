using System;
using System.Collections.Generic;

namespace CellUniverse.Models.Algorithms {

    using Infrastructure.Interfaces;

    public sealed class TheGameOfLifeSync : ICellAlgorithm {

        private short width, height;
        public bool[,] generation;
        private Random random;

        public TheGameOfLifeSync(short width, short height) {
            generation = new bool[width, height];
            Initialize();
            FillRandom();
        }

        public TheGameOfLifeSync(bool[,] generation) {
            this.generation = generation;
            width = (short)generation.GetLength(0);
            height = (short)generation.GetLength(1);
            Initialize();
        }

        private void Initialize() {
            random = new Random();
        }

        public bool IsIdentical(bool[,] layer) {
            for (int x = 0; x < generation.GetLength(0); x++) {
                for (int y = 0; y < generation.GetLength(1); y++) {
                    if (generation[x, y] != layer[x, y]) {
                        return false;
                    }
                }
            }
            return true;
        }

        private void FillRandom() {
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    generation[x, y] = random.Next(2) == 1;
                }
            }
        }

        IEnumerable<Tuple<short, short, bool>> ICellAlgorithm.NextGeneration() {

            var buffer2d = new bool[width, height];

            for (short x = 0; x < width; x++) {
                for (short y = 0; y < height; y++) {

                    int neighboursCount = CountNeighbours(generation, x, y);

                    if ((neighboursCount == 2 || neighboursCount == 3) && generation[x, y]) {
                        yield return new Tuple<short, short, bool>(x, y, buffer2d[x, y] = true);
                    }
                    if ((neighboursCount < 2 || neighboursCount > 3) && generation[x, y]) {
                        yield return new Tuple<short, short, bool>(x, y, buffer2d[x, y] = false);
                    }
                    if (neighboursCount == 3 && !generation[x, y]) {
                        yield return new Tuple<short, short, bool>(x, y, buffer2d[x, y] = true);
                    }
                }
            }
            generation = buffer2d;
        }

        private int CountNeighbours(bool[,] generation, short posX, short posY) {

            byte counter = 0;
            short startX = (short)(posX - 1), endX = (short)(posX + 2);
            short startY = (short)(posY - 1), endY = (short)(posY + 2);

            for (short x = startX; x < endX; x++) {
                for (short y = startY; y < endY; y++) {
                    if (x == posX && y == posY)
                        continue;

                    short px = x, py = y;

                    if (px == -1) px = (short)(width - 1);
                    else if (px == width) px = 0;

                    if (py == -1) py = (short)(height - 1);
                    else if (py == height) py = 0;

                    if (generation[px, py])
                        counter++;
                }
            }
            return counter;
        }      
    }
}
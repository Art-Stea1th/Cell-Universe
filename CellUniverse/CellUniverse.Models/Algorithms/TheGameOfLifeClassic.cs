namespace CellUniverse.Models.Algorithms {

    using Infrastructure.Interfaces;

    public sealed class TheGameOfLifeClassic : ICellAlgorithm {

        bool[,] ICellAlgorithm.NextGeneration(bool[,] prevGeneration) {

            var nextGeneration = new bool[prevGeneration.GetLength(0), prevGeneration.GetLength(1)];

            for (int y = 0; y < prevGeneration.GetLength(0); y++) {
                for (int x = 0; x < prevGeneration.GetLength(1); x++) {

                    int neighboursCount = CountNeighbours(prevGeneration, x, y);

                    if ((neighboursCount == 2 || neighboursCount == 3) && IsAlive(prevGeneration, x, y))
                        nextGeneration[y, x] = true;

                    if ((neighboursCount < 2 || neighboursCount > 3) && IsAlive(prevGeneration, x, y))
                        nextGeneration[y, x] = false;

                    if (neighboursCount == 3 && !IsAlive(prevGeneration, x, y))
                        nextGeneration[y, x] = true;
                }
            }
            return nextGeneration;
        }

        private int CountNeighbours(bool[,] generation, int posX, int posY) {

            int counter = 0;

            for (int y = posY - 1; y < posY + 2; y++) {
                for (int x = posX - 1; x < posX + 2; x++) {
                    if (x == posX && y == posY)
                        continue;

                    int py = y, px = x;

                    if (py == -1) py = generation.GetLength(0) - 1;
                    else if (py == generation.GetLength(0)) py = 0;

                    if (px == -1) px = generation.GetLength(1) - 1;
                    else if (px == generation.GetLength(1)) px = 0;

                    if (IsAlive(generation, px, py))
                        counter++;
                }
            }
            return counter;
        }

        private bool IsAlive(bool[,] generation, int posX, int posY) {
            return generation[posY, posX] ? true : false;
        }
    }
}
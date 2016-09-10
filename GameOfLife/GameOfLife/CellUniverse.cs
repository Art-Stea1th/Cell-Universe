using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;


namespace GameOfLife {

    public class CellUniverse {

        public bool[,] NewGeneration(bool[,] cells) {

            var nextGeneration = new bool[cells.GetLength(0), cells.GetLength(1)];

            for (int x = 0; x < cells.GetLength(0); x++) {
                for (int y = 0; y < cells.GetLength(1); y++) {
                    int neighbours = CountNeighbours(cells, x, y);

                    if ((neighbours == 2 || neighbours == 3) && cells[x, y])
                        nextGeneration[x, y] = true;

                    if ((neighbours < 2 || neighbours > 3) && cells[x, y])
                        nextGeneration[x, y] = false;

                    if (neighbours == 3 && !cells[x, y])
                        nextGeneration[x, y] = true;
                }
            }
            return nextGeneration;
        }

        private int CountNeighbours(bool[,] cells, int x, int y) {

            int counter = 0;

            int xFactor = x == 0 ? 0 : 1;
            int xFactorEnd = x == cells.GetLength(0)-1 ? 0 : 1;
            int yFactor = y == 0 ? 0 : 1;
            int yFactorEnd = y == cells.GetLength(1)-1 ? 0 : 1;

            for (int i = x - xFactor; i <= x + xFactorEnd; i++) {
                for (int n = y - yFactor; n <= y + yFactorEnd; n++) {
                    if (i == x && n == y)
                        continue;

                    if (IsAlive(cells, i, n))
                        counter++;
                }
            }
            return counter;
        }

        private static bool IsAlive(bool[,] cells, int i, int n) {
            try {
                if (cells[i, n])
                    return true;
            }
            catch {
                return false;
            }
            return false;
        }
    }
}
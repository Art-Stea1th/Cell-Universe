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

            for (int i = x - 1; i < x + 2; i++) {
                for (int j = y - 1; j < y + 2; j++) {
                    if (i == x && j == y)
                        continue;

                    int px = i;
                    int py = j;

                    if (px == -1) px = cells.GetLength(0) - 1;
                    else if (px == cells.GetLength(0)) px = 0;

                    if (py == -1) py = cells.GetLength(1) - 1;
                    else if (py == cells.GetLength(1)) py = 0;

                    if (IsAlive(cells, px, py))
                        counter++;
                }
            }
            return counter;
        }

        private static bool IsAlive(bool[,] cells, int i, int j) {
            try {
                if (cells[i, j])
                    return true;
            }
            catch {
                return false;
            }
            return false;
        }
    }
}
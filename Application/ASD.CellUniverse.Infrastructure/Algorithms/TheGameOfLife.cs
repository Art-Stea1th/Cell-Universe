using System.Threading.Tasks;

namespace ASD.CellUniverse.Infrastructure.Algorithms {

    using Interfaces;
    using MVVM;

    public sealed class TheGameOfLife : BindableBase, IMutationAlgorithm {

        private uint dead = 0, alive = (uint)255 << 24;

        public string Name => "Game Of Life";
        public override string ToString() => Name;

        public uint[,] Mutate(uint[,] prev) {

            var next = new uint[prev.GetLength(0), prev.GetLength(1)];

            Parallel.For(0, prev.GetLength(0), (x) => {
                Parallel.For(0, prev.GetLength(1), (y) => {

                    var neighbours = CountNeighbours(prev, x, y);

                    if ((neighbours == 2 || neighbours == 3) && prev[x, y] == alive) {
                        next[x, y] = alive;
                    }
                    else if ((neighbours < 2 || neighbours > 3) && prev[x, y] == alive) {
                        next[x, y] = dead;
                    }
                    else if (neighbours == 3 && prev[x, y] == dead) {
                        next[x, y] = alive;
                    }
                });
            });
            return next;
        }

        private int CountNeighbours(uint[,] cells, int x, int y) {

            int width = cells.GetLength(0), height = cells.GetLength(1);
            var count = 0;

            var l = x == 0 ? width - 1 : x - 1;
            var r = x == width - 1 ? 0 : x + 1;

            var t = y == 0 ? height - 1 : y - 1;
            var b = y == height - 1 ? 0 : y + 1;


            if (cells[l, t] == alive) { ++count; }
            if (cells[x, t] == alive) { ++count; }
            if (cells[r, t] == alive) { ++count; }

            if (cells[l, y] == alive) { ++count; }
            if (cells[r, y] == alive) { ++count; }

            if (cells[l, b] == alive) { ++count; }
            if (cells[x, b] == alive) { ++count; }
            if (cells[r, b] == alive) { ++count; }

            return count;
        }
    }
}
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;


namespace CellUniverse.Models.Algorithms {

    using Infrastructure.Interfaces;

    public sealed class TheGameOfLifeThreadPool : ICellAlgorithm {

        private short width, height;

        private bool[,] generation;
        private bool[,] buffer2d;

        private Random random;

        private ConcurrentQueue<Tuple<short, short, bool>> buffer;
        private int totalOfColumnsCalculate;

        private bool allColumnsBypassed;

        public TheGameOfLifeThreadPool(short width, short height) {
            generation = new bool[width, height];
            buffer2d = new bool[width, height];
            Initialize();
            FillRandom();
        }

        public TheGameOfLifeThreadPool(bool[,] generation) {
            this.generation = generation;
            width = (short)generation.GetLength(0);
            height = (short)generation.GetLength(1);
            buffer2d = new bool[width, height];
            Initialize();
        }

        private void Initialize() {
            random = new Random();
            buffer = new ConcurrentQueue<Tuple<short, short, bool>>();
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

            Reset();
            new Task(() => Calculate()).Start();

            while (!allColumnsBypassed || totalOfColumnsCalculate > 0 || buffer.Count > 0) {
                Tuple<short, short, bool> next = null;
                if (buffer.TryDequeue(out next)) {
                    yield return next;
                }
            }
            Swap();
        }

        private void Reset() {
            allColumnsBypassed = false;
        }

        private void Calculate() {

            for (short posX = 0; posX < width; ++posX) {
                Interlocked.Increment(ref totalOfColumnsCalculate);
                short column = posX;
                ThreadPool.QueueUserWorkItem(new WaitCallback(CalculateColumn), column);
            }
            allColumnsBypassed = true;
        }

        private void Swap() {
            //Buffer.BlockCopy(buffer2d, 0, generation, 0, width * height);
            generation = buffer2d;
            buffer2d = new bool[width, height];
        }

        private void CalculateColumn(object column) {

            short col = (short)column;

            for (short row = 0; row < height; row++) {

                short x = col, y = row;

                int neighboursCount = CountNeighbours(x, y);

                if ((neighboursCount == 2 || neighboursCount == 3) && generation[x, y]) {
                    buffer.Enqueue(new Tuple<short, short, bool>(x, y, buffer2d[x, y] = true));
                }
                if ((neighboursCount < 2 || neighboursCount > 3) && generation[x, y]) {
                    buffer.Enqueue(new Tuple<short, short, bool>(x, y, buffer2d[x, y] = false));
                }
                if (neighboursCount == 3 && !generation[x, y]) {
                    buffer.Enqueue(new Tuple<short, short, bool>(x, y, buffer2d[x, y] = true));
                }
            }
            Interlocked.Decrement(ref totalOfColumnsCalculate);
        }

        private byte CountNeighbours(short posX, short posY) {

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
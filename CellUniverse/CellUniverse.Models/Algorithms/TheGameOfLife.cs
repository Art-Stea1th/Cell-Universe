﻿using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;


namespace CellUniverse.Models.Algorithms {

    using Infrastructure.Interfaces;

    public sealed class TheGameOfLife : ICellAlgorithm {

        private int width, height;

        private bool[,] generation;
        private bool[,] buffer2d;

        private Random random;

        private ConcurrentQueue<Tuple<int, int, bool>> buffer;
        private int totalOfColumnsCalculate;

        private bool allColumnsBypassed;

        public TheGameOfLife(int width, int height) {
            generation = new bool[width, height];
            buffer2d = new bool[width, height];
            Initialize();
            FillRandom();
        }

        public TheGameOfLife(bool[,] generation) {
            this.generation = generation;
            width = generation.GetLength(0);
            height = generation.GetLength(1);
            buffer2d = new bool[width, height];
            Initialize();
        }

        private void Initialize() {
            random = new Random();
            buffer = new ConcurrentQueue<Tuple<int, int, bool>>();
        }

        private void FillRandom() {
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    generation[x, y] = random.Next(2) == 1;
                }
            }
        }

        IEnumerable<Tuple<int, int, bool>> ICellAlgorithm.NextGeneration() {

            Reset();
            new Task(() => Calculate()).Start();

            while (!allColumnsBypassed || totalOfColumnsCalculate > 0 || buffer.Count > 0) {
                Tuple<int, int, bool> next = null;
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

            for (int posX = 0; posX < width; ++posX) {
                Interlocked.Increment(ref totalOfColumnsCalculate);
                int column = posX;
                ThreadPool.QueueUserWorkItem(new WaitCallback(CalculateColumn), column);
            }
            allColumnsBypassed = true;
        }

        private void Swap() {
            generation = buffer2d;
            buffer2d = new bool[width, height];
        }

        private void CalculateColumn(object column) {

            int col = (int)column;

            for (int row = 0; row < height; row++) {

                int neighboursCount = CountNeighbours(col, row);

                if ((neighboursCount == 2 || neighboursCount == 3) && generation[col, row]) {
                    buffer.Enqueue(new Tuple<int, int, bool>(col, row, true));
                    buffer2d[col, row] = true;
                }
                if (neighboursCount == 3 && !generation[col, row]) {
                    buffer.Enqueue(new Tuple<int, int, bool>(col, row, true));
                    buffer2d[col, row] = true;
                }
            }
            Interlocked.Decrement(ref totalOfColumnsCalculate);
        }

        private byte CountNeighbours(int posX, int posY) {

            byte counter = 0;
            int startX = posX - 1, endX = posX + 1;
            int startY = posY - 1, endY = posY + 1;

            for (int x = startX; x <= endX; x++) {
                for (int y = startY; y <= endY; y++) {
                    if (x == posX && y == posY)
                        continue;

                    int px = x, py = y;

                    if (px == -1) px = width - 1;
                    else if (px == width) px = 0;

                    if (py == -1) py = height - 1;
                    else if (py == height) py = 0;

                    if (generation[px, py])
                        counter++;
                }
            }
            return counter;
        }
    }
}
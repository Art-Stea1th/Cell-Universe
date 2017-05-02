﻿using System;

namespace ASD.CellUniverse.Infrastructure.Algorithms {

    using Interfaces;
    using MVVM;

    public sealed class TheGameOfLife : BindableBase, IMutationAlgorithm {

        private uint threshold = (uint)190 << 24;

        public string Name => "The Game Of Life";
        public override string ToString() => Name;

        private Random random = new Random();

        public uint[,] Mutate(uint[,] prev) {
            return NextGeneration(prev);
        }

        public uint[,] NextGeneration(uint[,] cells) {

            var nextGeneration = new uint[cells.GetLength(0), cells.GetLength(1)];

            for (var x = 0; x < cells.GetLength(0); x++) {
                for (var y = 0; y < cells.GetLength(1); y++) {

                    var neighbours = CountNeighbours(cells, x, y);

                    if ((neighbours == 2 || neighbours == 3) && IsAlive(cells, x, y)) {
                        nextGeneration[x, y] = GetAlive();
                    }
                    else if ((neighbours < 2 || neighbours > 3) && IsAlive(cells, x, y)) {
                        nextGeneration[x, y] = GetDead();
                    }
                    else if (neighbours == 3 && !IsAlive(cells, x, y)) {
                        nextGeneration[x, y] = GetAlive();
                    }
                    else {
                        nextGeneration[x, y] = GetDead();
                    }
                }
            }
            return nextGeneration;
        }

        private uint GetDead() => (uint)(random.Next() % 24 + 16) << 24;
        private uint GetAlive() => (uint)(random.Next() % 64 + 190) << 24;

        private int CountNeighbours(uint[,] cells, int x, int y) {

            var counter = 0;

            for (var i = x - 1; i < x + 2; i++) {
                for (var j = y - 1; j < y + 2; j++) {

                    if (i == x && j == y) { continue; }

                    var px = i; var py = j;

                    if (px == -1) { px = cells.GetLength(0) - 1; }
                    else if (px == cells.GetLength(0)) { px = 0; }

                    if (py == -1) { py = cells.GetLength(1) - 1; }
                    else if (py == cells.GetLength(1)) { py = 0; }

                    if (IsAlive(cells, px, py)) {
                        counter++;
                    }
                }
            }
            return counter;
        }

        private bool IsAlive(uint[,] cells, int x, int y) => cells[x, y] >= threshold ? true : false;
    }
}
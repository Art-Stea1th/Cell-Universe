using System;
using System.Collections.Generic;
using System.Linq;

namespace ASD.CellUniverse.Infrastructure.Algorithms {

    using Interfaces;
    using MVVM;

    public class RandomMixer : BindableBase, IGenerationAlgorithm {

        public string Name => "Random Mixer";

        public override string ToString() => Name;

        private Random random;
        public RandomMixer() => random = new Random();

        public byte[,] GenerateNextBy(byte[,] prev) {

            var next = new byte[prev.GetLength(0), prev.GetLength(1)];

            var x = 0;
            foreach (var newX in RandomIndexesFrom(prev, 0)) {
                var y = 0;
                foreach (var newY in RandomIndexesFrom(prev, 1)) {
                    next[newX, newY] = prev[x, y];
                    ++y;
                }
                ++x;
            }
            return next;
        }

        private IEnumerable<int> RandomIndexesFrom(byte[,] field, int dimension) {
            var columnsIndexes = new List<int>(Enumerable.Range(0, field.GetLength(dimension)));
            while (columnsIndexes.Count > 0) {
                var i = random.Next(0, columnsIndexes.Count);
                yield return columnsIndexes[i];
                columnsIndexes.RemoveAt(i);
            }

        }
    }
}
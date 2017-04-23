using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ASD.CellUniverse.Infrastructure.Algorithms {

    using Interfaces;
    using MVVM;

    public sealed class TheGameOfLife : BindableBase, IGenerationAlgorithm {

        public string Name => "The Game Of Life";

        public override string ToString() => Name;

        public bool[,] GenerateNextBy(bool[,] prev) {
            return prev;
        }
    }
}

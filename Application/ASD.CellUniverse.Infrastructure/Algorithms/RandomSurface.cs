using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ASD.CellUniverse.Infrastructure.Algorithms {

    using Interfaces;
    using MVVM;

    public class RandomSurface : BindableBase, IGenerationAlgorithm {

        public bool[,] GenerateNextBy(bool[,] prev) {
            throw new NotImplementedException();
        }

        public RandomSurface() {

        }
    }
}

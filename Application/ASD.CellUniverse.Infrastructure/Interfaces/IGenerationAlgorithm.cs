using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IGenerationAlgorithm {

        string Name { get; }

        bool[,] GenerateNextBy(bool[,] prev);

    }
}
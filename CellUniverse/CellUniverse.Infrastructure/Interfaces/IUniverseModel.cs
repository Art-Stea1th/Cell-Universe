using System;
using System.Collections.Generic;

namespace CellUniverse.Infrastructure.Interfaces {

    public interface IUniverseModel {

        IEnumerable<Tuple<int, int, bool>> NextGeneration { get; }
    }
}
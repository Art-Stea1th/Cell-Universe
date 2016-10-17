using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;

namespace CellUniverse.Models.Factories.Universes {

    internal sealed class Managed : Universe {
        protected override Color[,] GetNextGeneration() {
            throw new NotImplementedException();
        }
    }
}
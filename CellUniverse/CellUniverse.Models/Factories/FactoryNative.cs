using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CellUniverse.Models.Factories.Universes;

namespace CellUniverse.Models.Factories {
    class FactoryNative : Factory {
        public override Universe CreateUniverse(int width, int height, int layersCount) {
            return new Native();
        }
    }
}

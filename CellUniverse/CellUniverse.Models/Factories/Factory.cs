using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace CellUniverse.Models.Factories {

    using Universes;

    public abstract class Factory {

        public abstract Universe CreateUniverse(int width, int height, int layersCount);
    }
}

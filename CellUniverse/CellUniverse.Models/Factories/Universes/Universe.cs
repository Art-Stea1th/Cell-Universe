using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;

namespace CellUniverse.Models.Factories.Universes {

    using FSM;

    public abstract class Universe {

        private IAutomat universeFsm;

        protected int width, height;
        protected int layersInGeneration;

        public virtual void StartSimulation() {
            universeFsm.Start();
        }
        public virtual void SuspendSimulation() {
            universeFsm.Halt();
        }
        public virtual void StopSimulation() {
            universeFsm.Terminate();
        }

        protected abstract Color[,] GetNextGeneration();
    }
}
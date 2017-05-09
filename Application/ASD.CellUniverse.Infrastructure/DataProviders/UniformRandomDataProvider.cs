using System;
using System.Security.Cryptography;

namespace ASD.CellUniverse.Infrastructure.DataProviders {

    internal sealed class UniformRandomDataProvider : IDisposable {

        private RNGCryptoServiceProvider crypto;

        public UniformRandomDataProvider() => crypto = new RNGCryptoServiceProvider();

        public byte NextByte() {
            var next = new byte[1];
            crypto.GetBytes(next);
            return next[0];
        }

        public uint NextUint() {
            var next = new byte[4];
            crypto.GetBytes(next);            
            return BitConverter.ToUInt32(next, 0);
        }

        public void Dispose() => crypto?.Dispose();
    }
}
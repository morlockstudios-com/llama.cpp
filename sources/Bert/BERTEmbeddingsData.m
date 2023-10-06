//
//  EmbeddingsData.m
//  
//
//  Created by Marc Terns on 9/23/23.
//

#import "BERTEmbeddingsData.h"

@implementation BERTEmbeddingsData

- (instancetype)initWithResourceURL:(NSURL *)resourceURL embeddings:(NSArray<NSArray<NSNumber *> *> *)embeddings {
    self = [super init];
    if (self) {
        _resourceURL = resourceURL;
        _embeddings = embeddings;
    }
    return self;
}

- (instancetype)initWithFileContents:(NSString *)fileContents embeddings:(NSArray<NSArray<NSNumber *> *> *)embeddings {
    self = [super init];
    if (self) {
        _fileContents = fileContents;
        _embeddings = embeddings;
    }
    return self;
}

@end

